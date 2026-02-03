# FERP XMP Metadata Contract v1.0

## Document Control

* **Title:** FERP XMP Metadata Contract
* **Version:** 1.0
* **Status:** Approved – implementation-ready
* **Scope:** Defines how FERP embeds and how consumers read stamp metadata in PDF files.

---

## 1) Purpose

This contract defines a standards-aligned mechanism for embedding **stamp metadata** into PDF files, including the administrator and agreement details used to generate stamps.

The metadata is intended to be:

* durable across third-party PDF processing tools,
* authoritative for automated systems.

---

## 2) Design Principles

1. **XMP is authoritative**
   All automated systems MUST rely on XMP metadata for correctness.

2. **PDF Info metadata is volatile**
   Standard PDF Info metadata fields are treated as presentation-only and may be dropped or rewritten by downstream tools.

3. **No semantic overloading**
   Document-level metadata standards (e.g., Dublin Core) are not repurposed to represent domain-specific music publishing data.

4. **Loss resistance over convenience**
   Metadata durability takes precedence over UI discoverability.

---

## 3) Standards and Conventions

* **XMP packet** embedded in the PDF catalog `/Metadata` stream
  * `/Type /Metadata`
  * `/Subtype /XML`
* **RDF container types** per XMP specification
* **UTF-8 text encoding**
* Namespace URIs are identifiers and are not required to resolve.

---

## 4) Namespace

### 4.1 Custom Namespace

* **Preferred XML prefix:** `ferp`
* **Namespace URI:**

  ```
  https://tulbox.app/ferp/xmp/1.0
  ```

### 4.2 Standard Namespaces Used

* RDF: `http://www.w3.org/1999/02/22-rdf-syntax-ns#`
* XMP container: `adobe:ns:meta/`

---

## 5) Property Definitions

### 5.1 Administrator

**Property:** `ferp:administrator`

* **Type:** `xsd:string`
* **Cardinality:** 1..1
* **Order:** N/A
* **Semantics:**
  The company name that administers the FERP stamping ruleset.
* **Authority:** **Authoritative** source for all automated processing.

#### Example

```xml
<ferp:administrator>TūlBOX Music Publishing</ferp:administrator>
```

---

### 5.2 Data Added Date

**Property:** `ferp:dataAddedDate`

* **Type:** `xsd:date`
* **Cardinality:** 1..1
* **Order:** N/A
* **Semantics:**
  The date the stamp metadata was generated and embedded.
* **Authority:** **Authoritative** source for audit and traceability.

#### Example

```xml
<ferp:dataAddedDate>2026-02-02</ferp:dataAddedDate>
```

---

### 5.3 Stamp Specification Version

**Property:** `ferp:stampSpecVersion`

* **Type:** `xsd:string`
* **Cardinality:** 1..1
* **Order:** N/A
* **Semantics:**
  The version of the stamp specification used to generate the stamp layout.
* **Authority:** **Authoritative** source for interpreting stamp layout rules.

#### Example

```xml
<ferp:stampSpecVersion>1.0</ferp:stampSpecVersion>
```

---

### 5.4 Agreements

**Property:** `ferp:agreements`

* **Type:** `rdf:Bag` of `rdf:li` (`rdf:Description`)
* **Cardinality:** 0..n
* **Order:** Unordered
* **Semantics:**
  Each bag entry represents a single stamp's agreement details. An agreement may contain multiple effective dates and territories for a shared publisher list.
* **Authority:** **Authoritative** source for all automated processing.

#### Agreement Entry Properties

Within each agreement entry (`rdf:Description`):

**Property:** `ferp:publishers`

* **Type:** `rdf:Bag` of `rdf:li` (`xsd:string`)
* **Cardinality:** 1..n
* **Order:** Unordered
* **Semantics:** List of controlled publishers for the stamp.

**Property:** `ferp:effectiveDates`

* **Type:** `rdf:Seq` of `rdf:li` (`rdf:Description`)
* **Cardinality:** 0..n
* **Order:** Oldest to newest
* **Semantics:** Each entry represents an effective date paired with its applicable territories.

**Property:** `ferp:territories`

* **Type:** `rdf:Bag` of `rdf:li` (`xsd:string`)
* **Cardinality:** 0..n
* **Order:** Alphabetical (ascending)
* **Semantics:** Controlled territories for the effective date entry.

---

## 6) Normalization Rules

Normalization MUST be applied before writing `ferp:administrator` and `ferp:agreements`.

For administrator and publisher/territory values:

1. Trim leading and trailing whitespace.
2. Collapse internal whitespace to a single ASCII space (`U+0020`).
3. Remove empty values.
4. De-duplicate by exact match after normalization (within each bag).
5. Preserve original casing.
6. Preserve punctuation.

For effective dates:

1. Use ISO 8601 date format `YYYY-MM-DD`.
2. Remove empty or invalid values.
3. De-duplicate by exact match.
4. Sort oldest to newest within each agreement.

For territories:

1. Apply the same whitespace normalization as publishers.
2. Remove empty values.
3. De-duplicate by exact match.
4. Sort alphabetically (ascending) within each effective date entry.

Recommended behavior: preserve first occurrence when de-duplicating.

---

## 7) Write Requirements

A compliant writer MUST:

1. Embed an XMP packet in the PDF catalog metadata stream.
2. Write `ferp:administrator`, `ferp:dataAddedDate`, `ferp:stampSpecVersion`, and
   `ferp:agreements` exactly as defined in Section 5.
3. Ensure XML is valid UTF-8 and properly escaped.

A compliant writer MAY:

* Preserve unrelated existing XMP content when updating metadata.

---

## 8) Read Requirements

A compliant reader MUST:

1. Read `ferp:administrator`, `ferp:dataAddedDate`, `ferp:stampSpecVersion`, and
   `ferp:agreements` from XMP if present.
2. Apply normalization rules after reading.

### Authority Rule

If multiple sources are present and differ:

* **XMP always wins**

---

## 9) Canonical Minimal XMP Example

```xml
<?xpacket begin='' id='W5M0MpCehiHzreSzNTczkc9d'?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description xmlns:ferp="https://tulbox.app/ferp/xmp/1.0">
      <ferp:administrator>Tulbox Music Publishing</ferp:administrator>
      <ferp:dataAddedDate>2026-02-02</ferp:dataAddedDate>
      <ferp:stampSpecVersion>1.0</ferp:stampSpecVersion>
      <ferp:agreements>
        <rdf:Bag>
          <rdf:li rdf:parseType="Resource">
            <ferp:publishers>
              <rdf:Bag>
                <rdf:li>Random Name Publishing Group</rdf:li>
                <rdf:li>Some Music Publishing Commpany</rdf:li>
              </rdf:Bag>
            </ferp:publishers>
            <ferp:effectiveDates>
              <rdf:Seq>
                <rdf:li rdf:parseType="Resource">
                  <ferp:date>2024-01-01</ferp:date>
                  <ferp:territories>
                    <rdf:Bag>
                      <rdf:li>Canada</rdf:li>
                      <rdf:li>United States</rdf:li>
                    </rdf:Bag>
                  </ferp:territories>
                </rdf:li>
                <rdf:li rdf:parseType="Resource">
                  <ferp:date>2024-06-15</ferp:date>
                  <ferp:territories>
                    <rdf:Bag>
                      <rdf:li>World (Excl. U.S.)</rdf:li>
                    </rdf:Bag>
                  </ferp:territories>
                </rdf:li>
              </rdf:Seq>
            </ferp:effectiveDates>
          </rdf:li>
        </rdf:Bag>
      </ferp:agreements>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end='w'?>
```

---

## 10) Versioning Policy

* Namespace `https://tulbox.app/ferp/xmp/1.0` is immutable.
* Any breaking change requires a new namespace version (e.g., `/2.0`).
* Older namespaces MUST continue to be supported for reading.

---

## 11) Compliance Summary

### Writers

* MUST write XMP `ferp:administrator` and `ferp:agreements`

### Readers

* MUST rely on XMP
